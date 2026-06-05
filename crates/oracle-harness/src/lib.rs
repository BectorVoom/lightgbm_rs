//! `oracle-harness` — the validation seam every parity test plugs into.
//!
//! Reads committed C++ golden fixtures and compares the Rust port against them.
//! The abs-diff comparator lands in Task 2; the RNG parity test and fixtures in
//! Task 3.
