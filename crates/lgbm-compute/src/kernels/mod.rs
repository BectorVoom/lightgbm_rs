//! `#[cube]` compute kernels (CMP-05).
//!
//! This plan (04-01) ships the minimal `construct_histograms` kernel that
//! settles the D-04a bit-determinism bet; `find_best_split` / `data_partition`
//! / `subtract_histograms` land in 04-02/04-03.

pub mod histogram;
