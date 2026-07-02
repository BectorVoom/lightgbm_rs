//! Phase-20 on-device score updater (§11, ODL-16) — **empty Wave-0 stub**.
//!
//! Owning phase: **20**. Filled by Plan **20-01** (Wave-2). Declared here as an
//! empty compiling module so the Wave-0 scaffolding plan (20-00) can register it in
//! `kernels/mod.rs` up front, letting Plan 20-01 fill exactly THIS file with no
//! same-wave `kernels/mod.rs` contention (the D-08 one-file-per-plan discipline used
//! for every Phase-14..19 kernel module).
//!
//! Scope when filled: the resident `cuda_score_` accumulator + `AddScoreConstant` /
//! `MultiplyScoreConstant` / the per-leaf `add_prediction_to_score` device ops and the
//! `boosting_on_cuda_`-keyed host-mirror toggle (`CopyFromCUDADeviceToHost`). Additive
//! and OFF by default behind the `LGBM_CUDA_ON_DEVICE` seam.
