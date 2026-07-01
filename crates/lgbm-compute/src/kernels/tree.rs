//! On-device tree mutation — `Split` / `Shrinkage` / `AddBias` (§10, ODL-14).
//!
//! **Wave-0 stub (18-01).** Declared now so the Wave-1 plan **18-03** can fill
//! exactly this one file without touching the shared `kernels/mod.rs`. It lands
//! as an empty compiling stub; 18-03 replaces this body with the flat device
//! `CUDATree` SoA arrays and the `SplitKernel` (14 field writes from
//! [`crate::kernels::split_info::SplitScalars`], NaN→0), ordered BEFORE the
//! partition step and returning `right_leaf_index` (§1/§10 hard invariant), plus
//! the elementwise `ShrinkageKernel` (`leaf_value *= rate`) and `AddBiasKernel`
//! (`leaf_value += val`). Additive and OFF by default behind `LGBM_CUDA_ON_DEVICE`
//! (D-13); scalar math stays f64, no f64 per-row hot loop (D-14).
