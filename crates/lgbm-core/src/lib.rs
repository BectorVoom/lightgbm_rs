//! `lgbm-core` — shared foundations for the pure-Rust LightGBM port.
//!
//! Holds the f32 numerical type/constant contract ([`types`]), the structured
//! domain errors at the crate boundary ([`error`]), and the bit-exact `Random`
//! LCG ([`random`]). Every later crate depends on this one.

pub mod error;
pub mod random;
pub mod types;
