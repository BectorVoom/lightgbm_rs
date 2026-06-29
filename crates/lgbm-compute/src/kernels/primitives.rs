//! Shared device primitives (CubeCL `#[cube]` kernels) — **stub**.
//!
//! Owning plans: filled by **14-03** (full-depth grow-loop subset) and **14-05**
//! (anchor-pinned skeletons). Scope locked by **ODL-01**, **D-01**, **D-02**.
//!
//! ## What lives here (when filled)
//! - **Prefix-sum** (inclusive/exclusive): within-plane via the cubecl-0.10
//!   `plane_inclusive_sum`/`plane_exclusive_sum` intrinsics; block-wide via a
//!   segmented `SharedMemory` LDS scan (RESEARCH Pattern 2); full-array via the
//!   3-kernel global scan with one reused `client.empty` scratch (Pattern 3).
//! - **Shuffle reductions** sum/max/min + dot-product via `plane_sum`/`plane_max`/
//!   `plane_min` (RESEARCH Pattern 1). [full depth — D-01]
//! - **Single-block bitonic argsort** (index-only, comparator/tie order mirrors the
//!   C++ `BitonicArgCompareKernel`; permutation asserted bit-exact). [full — D-01]
//! - **Percentile** (weighted/unweighted), **multi-block / `…Global` argsort**, and
//!   **`BitonicArgSortItems`** as anchor-pinned **skeletons** — correct + tested,
//!   depth finalized by their first consumer (Phase 19/22). [D-02]
//!
//! ## Analog file
//! `crates/lgbm-compute/src/kernels/histogram.rs` — the in-repo prior art for the
//! `#[cube]` + thin-launch-wrapper convention, the `SharedMemory`/`sync_cube()`
//! LDS idiom, the `Atomic<u64>` two's-complement fixed-point accumulator
//! (`SCALE_F32 = 2^30`; NEVER `Atomic<i64>`), and the `launch_unchecked` SAFETY
//! wrapper. The CPU anchor stays a **plain serial f64 fold** (D-10) — these
//! primitives are GPU-side, anchored against the C++/HIP fixtures (D-03).
//!
//! ## 14-01 de-risk input to consume (RESEARCH Open Q1 / Pitfall 1)
//! The Wave-0 smoke test (`tests/plane_intrinsic_smoke.rs`) RESOLVED the
//! plane-intrinsic lowering question:
//! - **cubecl-hip (gfx1100):** all four of `plane_inclusive_sum`,
//!   `plane_exclusive_sum`, `plane_max`, `plane_min` LOWER and match the serial
//!   reference within ~1e-6 — **no `plane_shuffle_up` manual-scan fallback is
//!   needed**.
//! - **cubecl-cpu:** the plane intrinsics (and the `UNIT_POS_PLANE` builtin) are
//!   **NOT supported** (`has_plane == false`, `plane_size == 1`). The cpu anchor
//!   therefore uses the **serial sequential fold** (`ReducePath::Sequential`), not
//!   a plane kernel. Route the numeric primitives that carry output through the
//!   `runtime::Capabilities::accumulate_type()` f64(cpu)/f32(hip) gate; plane
//!   collectives run on the `has_plane` (GPU) path only.
//!
//! ## Standard import preamble (mirror `histogram.rs:28-31` when filling)
//! ```ignore
//! use cubecl::prelude::*;
//! use crate::error::ComputeError;
//! use crate::runtime::ActiveRuntime;
//! ```
//!
//! Until 14-03/14-05 land, this module is an intentionally empty compiling stub
//! (no items) so the kernels barrel can expose it without `mod.rs` contention.
