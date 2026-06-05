//! `lgbm-dataset` — deterministic feature binning + immutable columnar dataset.
//!
//! Faithful 1:1 port of LightGBM's `src/io/` binning subsystem (D-04). Bin
//! boundaries and per-row bin indices are bit-identical to C++ (the determinism
//! root every downstream split inherits). Depends only on `lgbm-core`.
//!
//! This plan (Phase 2 Plan 01) delivers the first vertical slice: the numeric
//! `BinMapper::FindBin` + `ValueToBin` kernel and the `DatasetError` boundary
//! enum. Later plans add the `Bin` storage trait, categorical encoding, EFB,
//! metadata, the `Dataset` construct/finish-load pipeline, and ingestion; their
//! `pub mod` declarations are added here as they land.

pub mod bin;
pub mod bin_mapper;
pub mod dataset;
pub mod error;
pub mod feature_group;
pub mod ingest;
pub mod metadata;
pub mod multi_val_bin;

pub use bin::{Bin, BinValue, create_dense_bin, create_sparse_bin};
pub use bin_mapper::{BinMapper, BinType, MissingType};
pub use dataset::{Dataset, FinishedDataset};
pub use error::DatasetError;
pub use feature_group::FeatureGroup;
pub use ingest::{from_csc, from_csr, from_mat};
pub use metadata::Metadata;
pub use multi_val_bin::{MultiValBin, MultiValWidth};
