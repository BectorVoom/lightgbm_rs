---
quick_id: 260627-o6i
slug: resolve-cudabackend-runtime-error-gate-h
date: 2026-06-27
status: complete
commits:
  - 6bca8cc fix(quick-260627-o6i): gate host build_fix_scan/subtract_scan on a CpuBackend capability
---

# Summary: Resolve CudaBackend `build_fix_scan: not supported` runtime error

## Error

```
lightgbm_rs.LightGBMError: ... compute runtime error:
build_fix_scan: unified host build+fix+scan not supported on this backend
```

A direct follow-on to quick-260627-mpz: now that `device_type="cuda"` reaches the
CudaBackend, training hit this once a wide-enough leaf tripped the unified-path threshold.

## Root cause

The unified HOST fused paths `build_fix_scan` / `subtract_scan` are **CpuBackend-only**
(trait defaults error). The learner gated `smaller_unified`/`larger_unified` on
`!resident_eligible`, assuming *"not resident ⟹ CpuBackend"*. True for RocmBackend
(`resident_pool_supported()==true`), but **false for a GPU backend without a resident
pool** — CudaBackend/WgpuBackend are `!resident_eligible` yet can't run the host path,
so above the feature-count threshold they routed into the erroring default. (Below the
threshold they already used the working standard path — which is why it only failed
mid-training.)

## Fix (commit 6bca8cc)

- `crates/lgbm-compute/src/lib.rs`: new `Backend::host_unified_fused_supported()`
  (default `false`; `CpuBackend` overrides `true`); updated the two default-impl docs.
- `crates/lgbm-treelearner/src/learner.rs`: AND `backend.host_unified_fused_supported()`
  into both `smaller_unified` (1555) and `larger_unified` (1667) gates.

CpuBackend → byte-unchanged (still unified). RocmBackend → unchanged (already excluded
by resident). CudaBackend/WgpuBackend → fall through to the standard `construct_histograms`
+ batched `find_best_split` path (the same path their below-threshold leaves already use).

## Verified here (CPU host)

- `cargo build -p lgbm-compute -p lgbm-treelearner`: clean.
- `cargo test -p lgbm-compute --lib build_fix_scan`: 6/0.
- `cargo test -p lgbm-treelearner --lib`: 77/0 — **and** with
  `LGBM_UNIFIED_BFS_THRESHOLD=0 LGBM_UNIFIED_SUBSCAN_THRESHOLD=0` (forces the unified
  path through the learner on CpuBackend): 77/0 → CpuBackend unified path still exercised
  and bit-exact.
- `cargo test -p oracle-harness` (full default suite): green — boosting_parity 75/0,
  learner_parity 29/0, kernel_parity 7/0, raw_bin_train_parity 2/0 (C++ golden),
  config_drift 3/0, etc.
- `cargo check -p lgbm-compute --features cuda`: compiles (CudaBackend satisfies the
  Backend trait with the new method; cudarc loader needs no CUDA toolkit at check time).

## NOT verified here — NVIDIA runtime sign-off pending

No NVIDIA GPU / CUDA toolkit on this host. On Colab/Kaggle: rebuild
`maturin build --release --features cuda` and rerun the failing training — expect no
`build_fix_scan` error; CudaBackend trains via the standard path. Numerical fidelity
vs the CPU anchor should hold within the documented f32 GPU envelope.

## Out of scope

- A real device-resident / on-device fused path for CudaBackend (perf feature, not
  needed for correctness; untestable here).
- Any CpuBackend / RocmBackend behavior change.
