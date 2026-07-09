//! Tests for the ORA-01 abs-diff comparator (D-02, ~1e-6 tolerance).

use oracle_harness::comparator::{abs_diff_within, compare_within, Mismatch, ORACLE_TOL};

#[test]
fn abs_diff_within_boundary() {
    // |a - b| == tol → true (inclusive).
    assert!(abs_diff_within(1.0, 1.0 + 1e-6, 1e-6));
    // |a - b| <= tol → true.
    assert!(abs_diff_within(1.0, 1.0 + 5e-7, 1e-6));
    assert!(abs_diff_within(2.0, 2.0, 1e-6));
    // |a - b| > tol → false.
    assert!(!abs_diff_within(1.0, 1.0 + 2e-6, 1e-6));
    assert!(!abs_diff_within(0.0, 1.0, 1e-6));
}

#[test]
fn oracle_tol_is_1e_6() {
    assert_eq!(ORACLE_TOL, 1e-6_f32);
}

#[test]
fn compare_within_passes_for_close_vectors() {
    let rust = [1.0_f32, 2.0, 3.0];
    let cpp = [1.0_f32 + 5e-7, 2.0, 3.0 - 5e-7];
    assert!(compare_within(&rust, &cpp, ORACLE_TOL).is_ok());
}

#[test]
fn compare_within_reports_first_offending_index() {
    let rust = [1.0_f32, 2.0, 3.0, 4.0];
    let cpp = [1.0_f32, 2.0, 3.5, 4.5]; // index 2 is the first to diverge
    let err = compare_within(&rust, &cpp, ORACLE_TOL).unwrap_err();
    match err {
        Mismatch::ValueMismatch { index, .. } => assert_eq!(index, 2),
        other => panic!("expected ValueMismatch, got {other:?}"),
    }
}

#[test]
fn compare_within_reports_length_mismatch() {
    let rust = [1.0_f32, 2.0];
    let cpp = [1.0_f32, 2.0, 3.0];
    let err = compare_within(&rust, &cpp, ORACLE_TOL).unwrap_err();
    assert!(matches!(err, Mismatch::LengthMismatch { rust_len: 2, cpp_len: 3 }));
}
