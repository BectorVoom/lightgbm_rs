//! Numeric BinMapper golden parity — LAYER 2 (`numeric_assignment`).
//!
//! Replays the Rust `BinMapper::value_to_bin` over every row of every committed
//! numeric case and asserts the full per-row `u32` bin-index vector matches the
//! C++ golden EXACTLY (DAT-01 / SC#5), via `oracle_harness::comparator::
//! compare_exact_u32` (exact integer equality, NOT the `~1e-6` oracle tolerance).
//!
//! The corpus includes the curated edge columns (NaN-as-missing, +0.0/-0.0,
//! on-boundary, all-missing, single-value, all-zero, zero-as-missing,
//! pre-filter-trivial, dense). Reads ONLY the committed fixture; SKIPs gracefully
//! when absent (D-06).

mod golden;

use golden::Case;
use oracle_harness::comparator::{compare_exact_u32, Mismatch};

#[test]
fn numeric_assignment_matches_cpp_golden() {
    let Some(cases) = golden::load_cases() else {
        eprintln!(
            "numeric_assignment: SKIP — fixture not found. Run \
             `cargo run -p xtask -- bin-capture` and commit \
             tests/fixtures/numeric_binning.txt."
        );
        return;
    };
    assert!(!cases.is_empty(), "fixture must contain at least one case");

    let mut total_rows = 0usize;
    for case in &cases {
        let mapper = build_mapper(case);
        let got: Vec<u32> = case.column.iter().map(|&v| mapper.value_to_bin(v)).collect();

        match compare_exact_u32(&got, &case.golden_assign) {
            Ok(()) => {}
            Err(Mismatch::LengthMismatch { rust_len, cpp_len }) => panic!(
                "case `{}`: assignment length mismatch (rust={rust_len}, cpp={cpp_len})",
                case.name
            ),
            Err(Mismatch::ExactMismatch { index, rust, cpp }) => panic!(
                "case `{}`: row {index} value={} routed to bin {rust} but C++ golden is {cpp}",
                case.name,
                case.column[index],
            ),
            Err(other) => panic!("case `{}`: unexpected mismatch {other}", case.name),
        }
        total_rows += case.column.len();
    }

    eprintln!(
        "numeric_assignment: {} cases / {total_rows} rows replayed — per-row bin indices exact.",
        cases.len()
    );
}

fn build_mapper(case: &Case) -> lgbm_dataset::BinMapper {
    lgbm_dataset::BinMapper::find_bin_from_column(
        &case.column,
        case.max_bin,
        case.min_data_in_bin,
        case.min_split_data,
        case.pre_filter,
        case.use_missing,
        case.zero_as_missing,
        case.sample_cnt,
        case.data_random_seed,
        &[],
    )
}
