//! `oracle-harness` — the validation seam every parity test plugs into.
//!
//! Reads committed C++ golden fixtures and compares the Rust port against them.
//! The abs-diff comparator ([`comparator`]) proves the `~1e-6` float contract;
//! the RNG parity test (`tests/rng_parity.rs`) replays the committed randomized
//! golden set with no C++ toolchain at test time.

pub mod comparator;

pub use comparator::{abs_diff_within, compare_within, Mismatch, ORACLE_TOL};
