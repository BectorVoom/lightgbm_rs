//! On-device prediction — tree-walk `AddPredictionToScore` (§10, ODL-15).
//!
//! **Wave-0 stub (18-01).** Declared now so the Wave-2 plan **18-04** can fill
//! exactly this one file without touching the shared `kernels/mod.rs`. It lands
//! as an empty compiling stub; 18-04 replaces this body with the
//! `AddPredictionToScoreKernel<USE_INDICES>` tree-walk over the §13 columnar
//! store ([`crate::kernels::column_data`] / [`crate::kernels::row_data`] 8/16/32
//! dispatch), the shared numeric missing/default route (transcribed once, reused
//! by the partition mark — Pitfall 4), the categorical `FindInBitsetCUDA`
//! membership branch, and the §9 `AddPredictionToScoreKernel<USE_BAGGING>`
//! leaf-map gather-add. Additive and OFF by default behind `LGBM_CUDA_ON_DEVICE`
//! (D-13); `double` score accumulator only, no f64 per-row hot loop (D-14); the
//! objective inverse-link stays host-side at readback this phase (Phase-19).
