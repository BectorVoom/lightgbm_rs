---
quick_id: 260627-o6i
slug: resolve-cudabackend-runtime-error-gate-h
date: 2026-06-27
status: in-progress
---

# Quick Task: Resolve CudaBackend `build_fix_scan: not supported` runtime error

## Error (reported)

```
lightgbm_rs.LightGBMError: boosting error: tree learner error: compute backend error:
compute runtime error: build_fix_scan: unified host build+fix+scan not supported on this backend
```

Surfaced after quick-260627-mpz wired `device_type="cuda"` to the CudaBackend — so
the param now correctly reaches the GPU backend, which then hits this at training time.

## Root cause

The `Backend` trait has two CPU-only **host** fused paths — `build_fix_scan`
(`lib.rs:1128`) and `subtract_scan` (`lib.rs:1180`) — whose default impls return a
typed "not supported" error. **Only `CpuBackend` overrides them** (1519, 1555). The
`gpu_core_backend!` macro (CudaBackend/WgpuBackend, lib.rs:2001) overrides only the
4 device kernels and inherits the erroring defaults.

The learner gates these paths on `!resident_eligible`:
- `smaller_unified` (`learner.rs:1555`): `!smaller_fused && !self.resident_eligible && features.len() >= unified_bfs_threshold()`
- `larger_unified` (`learner.rs:1667`): `!self.resident_eligible && parent_slot.is_some() && … && features.len() >= unified_subscan_threshold()`

The trait docs (lib.rs:1117-1120, 1169-1172) state the assumption explicitly:
*"the gate ANDs in `!resident_pool_supported()` so this is never reached on a GPU
backend."* That assumed the **only** GPU backend (RocmBackend) has a resident pool
(`resident_pool_supported() == true`). `CudaBackend`/`WgpuBackend` are GPU backends
**without** a resident pool (the macro deliberately omitted the resident overrides,
commit 3c158be decision #2), so `resident_eligible`/`fused_eligible` are false →
above the feature-count threshold the learner wrongly routes them to the host
`build_fix_scan`/`subtract_scan` → typed error.

(Below-threshold leaves already take the standard build + batched-scan path and work
— the error only fires once a wide-enough leaf trips the unified threshold.)

## Fix

Add an explicit backend capability instead of inferring CPU from `!resident`:

1. **`crates/lgbm-compute/src/lib.rs`** — add a trait method
   `fn host_unified_fused_supported(&self) -> bool { false }` (default false),
   overridden to `true` in `impl Backend for CpuBackend`. Update the `build_fix_scan`
   / `subtract_scan` default-impl doc lines to reference this gate.
2. **`crates/lgbm-treelearner/src/learner.rs`** — AND `self.backend.host_unified_fused_supported()`
   into both `smaller_unified` (1555) and `larger_unified` (1667) gates.

Effect: CpuBackend (true) — byte-unchanged, still takes the unified path. RocmBackend —
already excluded by `resident_eligible`, unchanged. CudaBackend/WgpuBackend (false) —
fall through to the standard `construct_histograms` + batched `find_best_split` path
(both provided by the macro / trait default), which is the SAME path their
below-threshold leaves already use. No more error.

## Verification

- **Here (CPU):** `cargo build` + `cargo test -p lgbm-compute -p lgbm-treelearner`;
  the `LGBM_UNIFIED_BFS_THRESHOLD=0` forced-on bit-exact tests must still pass
  (CpuBackend `host_unified_fused_supported()==true` → unified path unchanged);
  oracle-harness bit-exact merge gate green.
- **NVIDIA host (user, Colab/Kaggle):** rebuild `maturin --features cuda`, rerun the
  failing training — no `build_fix_scan` error; trains on CudaBackend.

## Out of scope

- Giving CudaBackend a real resident pool / on-device fused path (perf feature, not
  needed for correctness; can't test here).
- Any CpuBackend / RocmBackend behavior change.
