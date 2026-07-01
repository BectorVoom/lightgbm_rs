//! On-device regression objectives — grad/hess + inverse-link (§5.1, ODL-05).
//!
//! Owning phase: **19** (ODL-05). Filled by **19-01**.
//!
//! ## What will live here (19-01)
//! The Rust `#[cube]` port of `src/objective/cuda/cuda_regression_objective.cu`
//! (§5.1 of `docs/cuda-kernel-design.md`) — the per-row
//! `GetGradientsKernel_Regression{L2,L1,Huber,Fair,Poisson,Quantile}<USE_WEIGHT>`
//! grad/hess kernels, the `ConvertOutputCUDAKernel_Regression{,_Poisson}`
//! inverse-link (sqrt/exp), and the `RenewTreeOutputCUDAKernel_Regression{L1,Quantile}`
//! per-leaf median refit.
//!
//! ## Anchor discipline (D-05)
//! The host `lgbm_objective::regression::Objective` grad/hess (the cpu f64 fold) is
//! the parity oracle — NEVER GPU-vs-GPU (D-05 / def-f8u-01). The default cubecl-cpu
//! f64 anchor exercises this module (D-08); it is additive and OFF by default behind
//! `LGBM_CUDA_ON_DEVICE` (D-06). Numerically faithful to
//! `crates/lgbm-objective/src/regression.rs`.
//!
//! Filled by 19-01 (empty compiling stub in Wave-1, 19-00).
#![allow(unused_imports)]

use cubecl::prelude::*;

use crate::error::ComputeError;
