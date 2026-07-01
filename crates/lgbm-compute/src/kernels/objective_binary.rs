//! On-device binary objective — sigmoid grad/hess + boost-from-score (§5.2, ODL-06).
//!
//! Owning phase: **19** (ODL-06). Filled by **19-02**.
//!
//! ## What will live here (19-02)
//! The Rust `#[cube]` port of `src/objective/cuda/cuda_binary_objective.cu`
//! (§5.2 of `docs/cuda-kernel-design.md`) — the per-row
//! `GetGradientsKernel_BinaryLogloss<USE_LABEL_WEIGHT,USE_WEIGHT>` sigmoid grad/hess,
//! `BoostFromScoreKernel_1/2_BinaryLogloss<USE_WEIGHT>` init-score reduction,
//! `ConvertOutputCUDAKernel_BinaryLogloss` sigmoid-probability inverse-link, and the
//! `ResetOVACUDALabelKernel` one-vs-all label rewrite (shared with multiclass OVA).
//!
//! ## Anchor discipline (D-05)
//! The host `lgbm_objective::binary::Binary` grad/hess (the cpu f64 fold) is the
//! parity oracle — NEVER GPU-vs-GPU (D-05 / def-f8u-01). The default cubecl-cpu f64
//! anchor exercises this module (D-08); it is additive and OFF by default behind
//! `LGBM_CUDA_ON_DEVICE` (D-06). Numerically faithful to
//! `crates/lgbm-objective/src/binary.rs`.
//!
//! Filled by 19-02 (empty compiling stub in Wave-1, 19-00).
#![allow(unused_imports)]

use cubecl::prelude::*;

use crate::error::ComputeError;
