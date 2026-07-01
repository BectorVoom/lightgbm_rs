//! On-device ranking objectives — lambdarank / rank_xendcg grad/hess (§5.4, ODL-08).
//!
//! Owning phase: **19** (ODL-08). Filled by **19-04**.
//!
//! ## What will live here (19-04)
//! The Rust `#[cube]` port of `src/objective/cuda/cuda_rank_objective.cu`
//! (§5.4 of `docs/cuda-kernel-design.md`) — the per-query
//! `GetGradientsKernel_LambdarankNDCG<…>{,_Sorted}` pairwise-lambda grad/hess and the
//! `GetGradientsKernel_RankXENDCG_{SharedMemory,GlobalMemory}` cross-entropy-NDCG
//! grad/hess (with the per-query `CUDARandom(objective_seed + q)` gamma draw).
//!
//! ## Anchor discipline (D-05)
//! The host `lgbm_objective::rank::{Lambdarank,RankXendcg}` grad/hess (the cpu f64
//! fold) is the parity oracle — NEVER GPU-vs-GPU (D-05 / def-f8u-01). The default
//! cubecl-cpu f64 anchor exercises this module (D-08); it is additive and OFF by
//! default behind `LGBM_CUDA_ON_DEVICE` (D-06). Anchored against the real
//! `lib_lightgbm` 4.6 `lambdarank_gh_iter{1,N}.txt` grad/hess goldens (D-01) and
//! numerically faithful to `crates/lgbm-objective/src/rank.rs`.
//!
//! Filled by 19-04 (empty compiling stub in Wave-1, 19-00).
#![allow(unused_imports)]

use cubecl::prelude::*;

use crate::error::ComputeError;
