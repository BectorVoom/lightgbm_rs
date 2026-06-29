//! `CUDARandom` device LCG (`#[cube]`) + host parity launcher — **stub**.
//!
//! Owning plan: filled by **14-04**. Scope locked by **ODL-02**, **D-04**.
//!
//! ## What lives here (when filled)
//! A `#[cube]` device LCG whose stream is **bit-identical** to the host `Random`,
//! plus a host launcher that seeds N tasks, draws K each, reads back, and is
//! asserted against the host draw sequence (D-04 — NOT a new C++ capture; the host
//! `Random` IS the oracle).
//!
//! ### Recurrence + draw methods to port (`crates/lgbm-core/src/random.rs:44-77`)
//! ```ignore
//! // x = 214013 * x + 2531011  — plain u32 ops; native hardware two's-complement
//! // wrap. NOT `wrapping_add`/`wrapping_mul` (those are host methods, not `#[cube]`
//! // intrinsics — RESEARCH Pitfall 2 / Assumption A2; verified end-to-end by the
//! // D-04 parity test, which is itself the proof of the device-wrap claim).
//! fn advance(state) -> u32 { state = state * 214013 + 2531011; state }
//! fn rand_int16(state) -> i32 { ((advance(state) >> 16) & 0x7FFF) as i32 }
//! fn rand_int32(state) -> i32 { (advance(state)        & 0x7FFFFFFF) as i32 }
//! fn next_float(state) -> f32 { rand_int16(state) as f32 / 32768.0 }
//! ```
//!
//! ### Parity assertion (D-04)
//! Seed identical; `RandInt16`/`RandInt32` compared as exact `i32`; `NextFloat`
//! compared via `f32::to_bits()` equality (exactly the host test idiom at
//! `random.rs:158-167` / `oracle-harness rng_parity.rs:107-115`). Anchor to the
//! host stream, never GPU-vs-GPU (D-10).
//!
//! ## Security V6 negative control (carry forward `random.rs:8-14`)
//! `CUDARandom`/`Random` are **non-cryptographic** deterministic PRNGs and MUST
//! NEVER be used as a security RNG. The device port inherits this constraint.
//!
//! ## Analog files
//! `crates/lgbm-core/src/random.rs` (the host `Random` LCG — the parity oracle) +
//! the `histogram.rs` host launcher shape (`create_from_slice` → `launch` →
//! `read_one_unchecked`).
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
