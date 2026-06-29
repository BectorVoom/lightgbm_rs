//! SoA pre-allocated device split-record (`CUDASplitInfo` analog) — **stub**.
//!
//! Owning plan: filled by **14-04**. Scope locked by **ODL-02**, **D-05**,
//! **D-06**, **D-08**.
//!
//! ## What lives here (when filled)
//! A **Struct-of-Arrays, host pre-allocated** device split-record: one CubeCL
//! `Handle` per field, each sized `[num_leaf_slots]`, allocated **once** in a
//! `new(client, num_leaf_slots)` constructor via `client.empty(...)` and indexed
//! by leaf-slot (RESEARCH Pattern 5). The C++ deep-copy `operator=` becomes a
//! "copy leaf-slot a → b across all field buffers"; for Phase-14 correctness a
//! host-side index copy is sufficient (RESEARCH A6 / D-07 defers the device
//! copy-kernel + 8/16-int readback packet to the Phase-17/18 consumer).
//!
//! ### Numeric fields now (D-06)
//! `is_valid`, `leaf_index`, `gain`, `inner_feature_index`, `threshold` (u32),
//! `default_left`, per-side `{left,right}_sum_gradients`/`_sum_hessians` (f64),
//! `_count` (i32), `_gain`, `_value` (f64), and the quantized
//! `{left,right}_sum_of_gradients_hessians` (i64, stored as `Atomic<u64>`
//! two's-complement bits — NEVER `Atomic<i64>`, RESEARCH Pitfall 2).
//!
//! ### Categorical fields RESERVED now (D-06)
//! `num_cat_threshold` (i32 `[num_leaf_slots]`), `cat_threshold` (u32) and
//! `cat_threshold_real` (i32) pre-allocated as reserved slabs sized
//! `[num_leaf_slots * MAX_CAT_PER_SPLIT]` — **Phase 22 fills them, not
//! restructures**. Define `MAX_CAT_PER_SPLIT` as a documented plain `usize` const,
//! Phase-22-tunable (mirror the `HIST_LDS_MAX` const convention; RESEARCH Open Q3
//! — resolved in 14-04 Task 1).
//!
//! ## Invariants
//! - **No per-split / per-`SplitInfo` device alloc anywhere (D-08)** — the C++
//!   `AllocateCatVectorsKernel` per-split `cudaMalloc` is exactly what CubeCL
//!   pre-allocation eliminates. The D-05/D-08 "alloc once, reuse" rule also covers
//!   any primitive scratch (e.g. the global-scan `num_blocks` buffer).
//!
//! ## Analog files
//! `crates/lgbm-compute/src/kernels/split.rs` (the `client.empty` pre-alloc) +
//! `crates/lgbm-compute/src/kernels/histogram.rs` (the resident-pool convention,
//! and the thin `#[cube(launch)]` wrapper at `histogram.rs:84-92` for the future
//! device slot-copy kernel `buf[b] = buf[a]`).
//!
//! ## Standard import preamble (mirror `histogram.rs:28-31` when filling)
//! ```ignore
//! use cubecl::prelude::*;
//! use crate::error::ComputeError;
//! use crate::runtime::ActiveRuntime;
//! ```
//!
//! Until 14-04 lands, this module is an intentionally empty compiling stub (no
//! items) so the kernels barrel can expose it without `mod.rs` contention.
