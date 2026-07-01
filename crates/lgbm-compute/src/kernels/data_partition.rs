//! On-device data partition — `mark → prefix-sum → scatter` (§9, ODL-13).
//!
//! **Wave-0 stub (18-01).** This module is declared now so the Wave-1 plan
//! **18-02** can fill exactly this one file without re-editing the shared
//! `kernels/mod.rs` (keeps the two Wave-1 plans on disjoint files). It lands as
//! an empty compiling stub; 18-02 replaces this body with the new §9-faithful
//! `GenDataToLeftBitVector` mark kernel, the `PrepareOffset`/`AggregateBlockOffset`
//! block-offset scans (built on [`crate::kernels::primitives`] u16/u32 launchers),
//! the `SplitInnerKernel` scatter, and the plain-stable-partition cpu f64 anchor
//! (D-04 CONFIRMED). Additive and OFF by default behind `LGBM_CUDA_ON_DEVICE`
//! (D-13); anchored to the cubecl-cpu f64 fold (D-12), never GPU-vs-GPU.
