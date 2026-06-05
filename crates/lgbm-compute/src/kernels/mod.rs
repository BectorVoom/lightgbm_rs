//! `#[cube]` compute kernels (CMP-05).
//!
//! Plan 04-01 shipped the minimal `construct_histograms` kernel that settled the
//! D-04a bit-determinism bet; 04-02 finalized it; 04-03 (this plan) adds
//! `find_best_split` (gain math in-kernel, D-01a), `data_partition` (stable
//! row->{left,right} routing), and `subtract_histograms` (the kernel-layer
//! histogram-subtraction math, A3 resolved in-scope).

pub mod histogram;
pub mod partition;
pub mod split;
pub mod subtract;
