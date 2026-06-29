//! Device dataset (CUDARowData / CUDAColumnData) host-vs-device parity — ODL-03.
//!
//! ## Anchor discipline (D-08 / D-10)
//! Every device re-lay is asserted **bit-for-bit** against the host [`BinColumn`]
//! binned-value oracle (the v1.0 Rust binning is already C++-bit-exact — the same
//! Phase-14 D-04 reasoning), run ONLY on the cpu f64 anchor ([`cpu_client`]) —
//! NEVER GPU-vs-GPU. Bins are integer indices, so equality is exact `assert_eq!`
//! (no float tolerance).
//!
//! ## D-04 sparse synthesizer ([`synth`])
//! Synthesizes sparse columns whose logical nnz crosses 2^16 and 2^32 so each
//! `row_ptr_type` width {16, 32, 64} is forced, plus a column over the
//! `shared_hist_size / 2` budget for the large-bin spill (Pitfall 1). The nnz is
//! carried as a number (not materialized) so the 64-bit cell is exercisable without
//! a 2^32-element allocation.
//!
//! These tests panic at the Wave-0 `todo!()` stub bodies until Plans 15-02/15-03
//! land — that is expected this wave (they compile + link + run, not pass).

use lgbm_compute::kernels::column_data::{ColumnFeatureMeta, CudaColumnData};
use lgbm_compute::kernels::row_data::{divide_cuda_feature_groups, CudaRowData};
use lgbm_compute::runtime::{cpu_client, ActiveRuntime};
use lgbm_compute::BinColumn;

/// D-04 sparse-column synthesizer. Each entry carries the dense binned values (the
/// host oracle) plus the logical nnz that selects the CSR `row_ptr_type` width.
mod synth {
    use lgbm_compute::BinColumn;

    /// One synthesized column: the host binned-value oracle + the metadata that
    /// drives device width selection.
    pub struct SynthColumn {
        /// Dense binned values — the parity oracle (host truth).
        pub bins: BinColumn,
        /// The feature's bin count (selects `bit_type` ∈ {8, 16, 32}).
        pub num_bin: u32,
        /// Logical non-zeros — selects the CSR `row_ptr_type` width {16, 32, 64}
        /// (carried as a number; NOT materialized — D-04).
        pub nnz: u64,
    }

    /// The CSR `row_ptr_type` width forced by a logical nnz: 16-bit if it fits a
    /// `u16` row-pointer, 32-bit if a `u32`, else 64-bit.
    #[must_use]
    pub fn row_ptr_type_for(nnz: u64) -> u32 {
        if nnz <= u64::from(u16::MAX) {
            16
        } else if nnz <= u64::from(u32::MAX) {
            32
        } else {
            64
        }
    }

    const NUM_ROWS: usize = 8;

    fn col(values: &[u32], num_bin: u32, nnz: u64) -> SynthColumn {
        SynthColumn {
            bins: BinColumn::new(values.to_vec(), num_bin),
            num_bin,
            nnz,
        }
    }

    /// Synthesize the parity corpus: one column per `bit_type` width, each tagged
    /// with an nnz that forces a distinct `row_ptr_type` (16/32/64), plus a
    /// large-bin column whose own bin count exceeds the SMEM budget (the Pitfall-1
    /// spill). All columns share `NUM_ROWS` rows so they re-lay as one matrix.
    #[must_use]
    pub fn synth_columns() -> Vec<SynthColumn> {
        // u8 column (num_bin <= 256), small nnz -> row_ptr_type 16.
        let c_u8 = col(&[0, 1, 2, 3, 0, 1, 2, 3], 200, 1_000);
        // u16 column (256 < num_bin <= 65536), nnz crosses 2^16 -> row_ptr_type 32.
        let c_u16 = col(&[0, 300, 301, 5, 6, 7, 8, 9], 4_000, 70_000);
        // u32 column (num_bin > 65536), nnz crosses 2^32 -> row_ptr_type 64.
        let c_u32 = col(&[0, 70_000, 1, 2, 3, 4, 5, 6], 100_000, 5_000_000_000);
        // Large-bin spill column: its own bin count exceeds the budget (3072 for
        // shared_hist_size 6144) -> its own large-bin partition (Pitfall 1).
        let c_large = col(&[0, 1, 2, 3, 4, 5, 6, 4000], 4_096, 2_000);
        vec![c_u8, c_u16, c_u32, c_large]
    }

    /// The number of rows every synthesized column shares.
    #[must_use]
    pub fn num_rows() -> usize {
        NUM_ROWS
    }
}

/// ODL-03: dense re-lay reproduces every (row, column) bin at each `bit_type`
/// width (u8/u16/u32). Anchor: host [`BinColumn::bin`].
#[test]
fn dense_bin_parity_all_widths() {
    let client = cpu_client();

    // One column per width (num_bin selects U8 / U16 / U32).
    let columns = vec![
        BinColumn::new(vec![0, 1, 2, 3, 0, 1], 200),       // U8
        BinColumn::new(vec![0, 300, 1, 2, 400, 5], 4_000), // U16
        BinColumn::new(vec![0, 70_000, 1, 2, 3, 4], 100_000), // U32
    ];
    let num_rows = columns[0].len();
    let num_bin_per_column: Vec<usize> = columns.iter().map(|_| 0usize).collect();
    // hand layout: pack all columns into partitions under a generous budget.
    let layout = divide_cuda_feature_groups(&num_bin_per_column, 6144);

    let device =
        CudaRowData::<ActiveRuntime>::get_dense_data_partitioned(&client, &columns, &layout)
            .expect("dense re-lay");

    for (col_idx, column) in columns.iter().enumerate() {
        for row in 0..num_rows {
            let host = column.bin(row);
            let dev = device.read_bin(&client, row, col_idx).expect("read back");
            assert_eq!(dev, host, "dense bin mismatch at (row={row}, col={col_idx})");
        }
    }
}

/// ODL-03: `DivideCUDAFeatureGroups` partition layout vs hand-computed
/// expectations, including the Pitfall-1 large-bin spill.
#[test]
fn feature_partition_layout() {
    // shared_hist_size 6144 -> budget (max_num_bin_per_partition) = 3072.
    // Columns: [1000, 1000, 1000] pack (3000 <= 3072 would overflow at the 3rd;
    // the 3rd opens a new partition), and a 4th column of 4000 > 3072 spills to its
    // OWN large-bin partition.
    let num_bin_per_column = vec![1000usize, 1000, 1000, 4000];
    let layout = divide_cuda_feature_groups(&num_bin_per_column, 6144);

    assert_eq!(layout.shared_hist_size, 6144);
    // Hand-computed: partition 0 = {col0, col1} (2000 <= 3072), partition 1 = {col2}
    // (3000 > 3072 starts a new partition), partition 2 = {col3} large-bin spill.
    assert_eq!(layout.num_feature_partitions, 3, "partition count");
    assert_eq!(
        layout.feature_partition_column_index_offsets,
        vec![0, 2, 3, 4],
        "partition i owns columns [off[i], off[i+1])"
    );
    // column_hist_offsets are partition-local: col0@0, col1@1000 (partition 0);
    // col2@0 (partition 1); col3@0 (partition 2).
    assert_eq!(
        layout.column_hist_offsets,
        vec![0, 1000, 0, 0],
        "per-column partition-local bin offsets"
    );
    // partition_hist_offsets are the global bin offsets per partition begin.
    assert_eq!(
        layout.partition_hist_offsets,
        vec![0, 2000, 3000, 7000],
        "global partition begin offsets"
    );
    // The 4000-bin column exceeds the 3072 budget -> exactly one large-bin partition.
    assert_eq!(layout.num_large_bin_partition, 1, "Pitfall 1 large-bin spill");
}

/// ODL-03: sparse re-lay over the (bit_type, row_ptr_type) 3×3 matrix — each
/// `row_ptr_type` width is forced, a 2-partition case re-lays partition-local
/// bins (Pitfall 4), and an unsupported width is a typed error (Pitfall 2).
#[test]
fn sparse_relay_3x3_and_partition_local() {
    let client = cpu_client();
    let cols = synth::synth_columns();
    let bin_columns: Vec<BinColumn> = cols.iter().map(|c| c.bins.clone()).collect();
    let num_rows = synth::num_rows();

    let num_bin_per_column: Vec<usize> =
        cols.iter().map(|c| c.num_bin as usize).collect();
    let layout = divide_cuda_feature_groups(&num_bin_per_column, 6144);

    // (a) every row_ptr_type {16, 32, 64} cell is exercised by the synthesizer.
    let mut widths_seen = std::collections::BTreeSet::new();
    for c in &cols {
        let rpt = synth::row_ptr_type_for(c.nnz);
        widths_seen.insert(rpt);
        let device = CudaRowData::<ActiveRuntime>::get_sparse_data_partitioned(
            &client,
            &bin_columns,
            &layout,
            rpt,
        )
        .expect("sparse re-lay");
        assert_eq!(device.row_ptr_bit_type(), rpt, "row_ptr width selected");

        // (b) partition-local re-lay (Pitfall 4): the stored partition-local bin
        // equals the logical global bin minus the partition's global offset.
        for col_idx in 0..bin_columns.len() {
            let partition = (0..layout.num_feature_partitions)
                .find(|&p| {
                    let lo = layout.feature_partition_column_index_offsets[p];
                    let hi = layout.feature_partition_column_index_offsets[p + 1];
                    (lo..hi).contains(&col_idx)
                })
                .expect("column belongs to a partition");
            let local_column =
                col_idx - layout.feature_partition_column_index_offsets[partition];
            for row in 0..num_rows {
                let global = device.read_bin(&client, row, col_idx).expect("global");
                let local = device
                    .read_partition_local_bin(&client, partition, row, local_column)
                    .expect("partition-local");
                let part_off = layout.partition_hist_offsets[partition] as u32;
                assert_eq!(
                    local,
                    global - part_off,
                    "partition-local bin = global - partition_hist_offsets[{partition}]"
                );
            }
        }
    }
    assert_eq!(
        widths_seen,
        [16u32, 32, 64].into_iter().collect(),
        "all three row_ptr_type widths forced by the synthesizer"
    );

    // (c) an unsupported row_ptr width is a typed error, NOT a silent widen
    // (Pitfall 2).
    let err = CudaRowData::<ActiveRuntime>::get_sparse_data_partitioned(
        &client,
        &bin_columns,
        &layout,
        8, // unsupported
    );
    assert!(err.is_err(), "unsupported row_ptr width must error (Pitfall 2)");
}

/// ODL-03: column-major store read-back + numeric `ColumnFeatureMeta` parity.
#[test]
fn column_store_parity() {
    let client = cpu_client();

    let columns = vec![
        BinColumn::new(vec![0, 1, 2, 3, 0, 1], 200),       // U8
        BinColumn::new(vec![0, 300, 1, 2, 400, 5], 4_000), // U16
    ];
    let feature_meta = vec![
        ColumnFeatureMeta {
            bit_type: 8,
            feature_min_bin: 0,
            feature_max_bin: 3,
            offset: 0,
            most_freq_bin: 0,
            default_bin: 0,
            missing_is_zero: false,
            missing_is_na: false,
            feature_to_column: 0,
        },
        ColumnFeatureMeta {
            bit_type: 16,
            feature_min_bin: 0,
            feature_max_bin: 400,
            offset: 200,
            most_freq_bin: 0,
            default_bin: 0,
            missing_is_zero: false,
            missing_is_na: false,
            feature_to_column: 1,
        },
    ];

    let device =
        CudaColumnData::<ActiveRuntime>::new(&client, &columns, feature_meta.clone())
            .expect("column upload");

    for (col_idx, column) in columns.iter().enumerate() {
        let dev = device.read_column(&client, col_idx).expect("read column");
        assert_eq!(
            dev,
            column.to_u32_vec(),
            "column {col_idx} binned values mismatch"
        );
        // numeric meta round-trips.
        assert_eq!(
            device.feature_meta[col_idx], feature_meta[col_idx],
            "ColumnFeatureMeta mismatch for column {col_idx}"
        );
    }
}
