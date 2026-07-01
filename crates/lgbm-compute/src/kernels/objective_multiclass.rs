//! On-device multiclass objective — softmax grad/hess + per-row softmax (§5.3, ODL-07).
//!
//! Owning phase: **19** (ODL-07). Filled by **19-03**.
//!
//! ## What will live here (19-03)
//! The Rust `#[cube]` port of `src/objective/cuda/cuda_multiclass_objective.cu`
//! (§5.3 of `docs/cuda-kernel-design.md`) — the class-major
//! `GetGradientsKernel_MulticlassSoftmax<USE_WEIGHT>` grad/hess and the
//! `ConvertOutputCUDAKernel_MulticlassSoftmax` per-row softmax-probability
//! inverse-link. Multiclass-OVA reuses the binary per-class kernels (§5.2).
//!
//! ## Anchor discipline (D-05)
//! The host `lgbm_objective::multiclass` grad/hess (the cpu f64 fold) is the parity
//! oracle — NEVER GPU-vs-GPU (D-05 / def-f8u-01). The default cubecl-cpu f64 anchor
//! exercises this module (D-08); it is additive and OFF by default behind
//! `LGBM_CUDA_ON_DEVICE` (D-06). Numerically faithful to
//! `crates/lgbm-objective/src/multiclass.rs`.
//!
//! Filled by 19-03 (empty compiling stub in Wave-1, 19-00).
#![allow(unused_imports)]

use cubecl::prelude::*;

use crate::error::ComputeError;
