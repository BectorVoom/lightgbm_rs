//! Numeric BinMapper golden parity — LAYER 1 (`bin_mapper_internals`).
//!
//! Replays the Rust `BinMapper::find_bin_from_column` over EVERY committed
//! numeric case and asserts the bin-mapper internals match the C++ golden
//! bit-exact (DAT-01, ORA-03):
//!   - `bin_upper_bound_` f64 array — compared **bit-exact** via `.to_bits()`
//!     (NOT the `~1e-6` oracle tolerance; a 1-ULP boundary drift is a real
//!     divergence),
//!   - `num_bin`, `bin_type`, `missing_type`, `default_bin`, `most_freq_bin`,
//!     `is_trivial`.
//!
//! Reads ONLY the committed fixture (`tests/fixtures/numeric_binning.txt`); needs
//! NO C++ toolchain (D-06). Until `cargo run -p xtask -- bin-capture` is run and
//! the fixture committed, the file is absent and the test reports a SKIP so
//! `cargo test` stays green pre-capture.

mod golden;

use golden::{Case, MissingTypeCode};
use lgbm_dataset::{BinType, MissingType};

#[test]
fn bin_mapper_internals_match_cpp_golden() {
    let Some(cases) = golden::load_cases() else {
        eprintln!(
            "bin_mapper_internals: SKIP — fixture not found. Run \
             `cargo run -p xtask -- bin-capture` on a machine with a C++ toolchain \
             and commit tests/fixtures/numeric_binning.txt."
        );
        return;
    };
    assert!(!cases.is_empty(), "fixture must contain at least one case");

    for case in &cases {
        let mapper = build_mapper(case);

        // --- layer 1: bin_upper_bound_ bit-exact ------------------------------
        assert_eq!(
            mapper.bin_upper_bound_.len(),
            case.golden_upper.len(),
            "case `{}`: num upper bounds differ (rust={}, cpp={})",
            case.name,
            mapper.bin_upper_bound_.len(),
            case.golden_upper.len()
        );
        for (i, (&got, &exp)) in mapper
            .bin_upper_bound_
            .iter()
            .zip(case.golden_upper.iter())
            .enumerate()
        {
            assert_eq!(
                got.to_bits(),
                exp.to_bits(),
                "case `{}`: bin_upper_bound_[{i}] bit mismatch \
                 (rust={got} bits=0x{:016X}, cpp={exp} bits=0x{:016X})",
                case.name,
                got.to_bits(),
                exp.to_bits()
            );
        }

        // --- scalar meta ------------------------------------------------------
        assert_eq!(
            mapper.num_bin_, case.golden_num_bin,
            "case `{}`: num_bin mismatch",
            case.name
        );
        let exp_bin_type = match case.golden_bin_type {
            0 => BinType::Numerical,
            _ => BinType::Categorical,
        };
        assert_eq!(
            mapper.bin_type_, exp_bin_type,
            "case `{}`: bin_type mismatch",
            case.name
        );
        let exp_missing = match case.golden_missing_type {
            MissingTypeCode::None => MissingType::None,
            MissingTypeCode::Zero => MissingType::Zero,
            MissingTypeCode::NaN => MissingType::NaN,
        };
        assert_eq!(
            mapper.missing_type_, exp_missing,
            "case `{}`: missing_type mismatch",
            case.name
        );
        assert_eq!(
            mapper.default_bin_, case.golden_default_bin,
            "case `{}`: default_bin mismatch",
            case.name
        );
        assert_eq!(
            mapper.most_freq_bin_, case.golden_most_freq_bin,
            "case `{}`: most_freq_bin mismatch",
            case.name
        );
        assert_eq!(
            mapper.is_trivial_, case.golden_is_trivial,
            "case `{}`: is_trivial mismatch",
            case.name
        );
    }

    eprintln!(
        "bin_mapper_internals: {} cases replayed — bin_upper_bound_ + meta bit-for-bit.",
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
