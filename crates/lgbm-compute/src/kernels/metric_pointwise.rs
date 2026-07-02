//! Phase-20 on-device pointwise metric evaluator (§12, ODL-17) — **empty Wave-0 stub**.
//!
//! Owning phase: **20**. Filled by Plan **20-02** (Wave-2). Declared here as an empty
//! compiling module so the Wave-0 scaffolding plan (20-00) can register it in
//! `kernels/mod.rs` up front, letting Plan 20-02 fill exactly THIS file with no
//! same-wave `kernels/mod.rs` contention (the D-08 one-file-per-plan discipline used
//! for every Phase-14..19 kernel module).
//!
//! Scope when filled: the `EvalKernel` + two-stage f64 reduction over the 12 pointwise
//! (regression/binary) losses the [`crate::device_metric::metric_supported`]
//! discriminator gates on-device, composing `objective_regression::convert_output_on`
//! for the ConvertOutput inverse-link and `primitives::reduce_sum_f64_on` for the
//! reduction; anchor-pinned to the real `lib_lightgbm` goldens under
//! `oracle-harness/tests/fixtures/metric/`. Additive and OFF by default behind the
//! `LGBM_CUDA_ON_DEVICE` seam.
